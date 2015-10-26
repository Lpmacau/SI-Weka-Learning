package aula6;

import java.util.Enumeration;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

public class LeitorDataset {
	
	

	public static void main(String[] args) throws Exception {
		String path = "/Users/Periodico/Desktop/database.csv";
		
		// Ler dataset e criar o Instances (permite o acesso aos dados)
		DataSource dataset = new DataSource(path);
		Instances instances = dataset.getDataSet();
		
		System.out.println(" --------- DATASET ---------");
		
		// Atributos do dataset
		for(int i = 0; i<instances.numAttributes(); i++){
			Attribute a = instances.attribute(i);
			System.out.println("Atributo "+i+": "+a.name());
		}
		
		
		System.out.println("----------------------------------------------");
		
		// Valor das instancias
		for(int i = 0; i<10; i++){
			Instance e = instances.instance(i);
			System.out.println("Instância "+i+": "+e);
		}	
		
		System.out.println("----------------------------------------------");
		
		
		// Aplicação de um Filtro (Remover a primeira coluna)
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "1";
		
		Remove remove = new Remove();
		remove.setOptions(options);
		remove.setInputFormat(instances);
		
		Instances newData = Filter.useFilter(instances, remove);
		
		
		// Valor das instâncias após a aplicação do filtro
		for(int i = 0; i<10; i++){
			Instance e = newData.instance(i);
			System.out.println("Instância "+i+" Remove: "+e);
		}
		
		System.out.println("----------------------------------------------");
		
		// Treino de um Dataset
		String[] optionsTrain = new String[1];
		optionsTrain[0] = "-U";
		
		// Aplicação de um filtro de Discretize (para o ultimo atributo deixar de ser numerico e passe a ser nominal)
		String[] optionsFiltro = new String[6];
		optionsFiltro[0] = "-B";
		optionsFiltro[1] = "10";
		optionsFiltro[2] = "-M";
		optionsFiltro[3] = "-1.0";
		optionsFiltro[4] = "-R";
		optionsFiltro[5] = "last";
		
		Discretize d = new Discretize();
		d.setOptions(optionsFiltro);
		d.setInputFormat(instances);
		
		Instances discretizado = Filter.useFilter(instances, d);
		discretizado.setClassIndex(instances.numAttributes() - 1); 
		
		for(int i = 0; i<10; i++){
			Instance e = discretizado.instance(i);
			System.out.println("Instância "+i+" Discretizado: "+e);
		}
		
		System.out.println("----------------------------------------------");

		// Árvore de decisão
		J48 tree = new J48();
		tree.setOptions(optionsTrain);
		tree.buildClassifier(discretizado);
		
		// Classificação de uma instância
		for(int i = 0; i<10; i++){
			Instance e = discretizado.instance(i);
			double valor = tree.classifyInstance(e);
			System.out.println("Instância "+i+" DecisionTree (valor): "+valor);

		}
		
		System.out.println("----------------------------------------------");
		
		
		// CLUSTERING
		String[] optionsKMeans = new String[2];
		optionsKMeans[0] = "-I";
		optionsKMeans[1] = "100";
		
		SimpleKMeans clusterer = new SimpleKMeans();
		clusterer.setOptions(optionsKMeans);
		clusterer.buildClusterer(instances);
		
		for(int i = 0; i<10; i++){
			Instance e = instances.instance(i);
			double valor = clusterer.clusterInstance(e);
			System.out.println("Instância "+i+" Clustering (valor): "+valor);

		}
		
		System.out.println("----------------------------------------------");

		
		// Rule Creation
		String[] optionsRules = new String[2];
		optionsRules[0] = "-N";
		optionsRules[1] = "10";
		

		String[] optionsFiltro2 = new String[4];
		optionsFiltro2[0] = "-B";
		optionsFiltro2[1] = "10";
		optionsFiltro2[2] = "-M";
		optionsFiltro2[3] = "-1.0";

		
		Discretize d2 = new Discretize();
		d.setOptions(optionsFiltro2);
		d.setInputFormat(instances);
		
		Instances discretizado2 = Filter.useFilter(instances, d); 
		
		Apriori ap = new Apriori();
		ap.setOptions(optionsRules);
		ap.buildAssociations(discretizado2);
		
		AssociationRules ar = ap.getAssociationRules();
		for(AssociationRule regra : ar.getRules()){
			System.out.println(regra.toString());
		}
		
		
		
		
		
		
	}

}
